import { Component, OnInit , ViewChild } from '@angular/core';
import { User } from 'src/app/service/Interfaces/user';
import { ProjetService } from '../../service/ImplementationService/projet.service'
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { HttpClient } from '@angular/common/http'
import { Router } from '@angular/router'
import  Swal  from 'sweetalert2';
@Component({
  selector: 'app-confirmation',
  templateUrl: './confirmation.component.html',
  styleUrls: ['./confirmation.component.css']
})
export class ConfirmationComponent implements OnInit {
  users:User[]
  user:User
  form: FormGroup;
  
  constructor(private projetService: ProjetService, private _router: Router, private http: HttpClient,private formBuilder: FormBuilder) { }

  ngOnInit(): void {
    this.getUsers();
    
    this.form = this.formBuilder.group({
      email: [null, [Validators.required]],
      username: [null, Validators.required],
      firstname: [null, Validators.required],
      lastname: [null, Validators.required],
      address: [null, Validators.required],
      num_phone: [null, Validators.required],
      city: [null, Validators.required],
      country: [null, Validators.required],
      password: [null, [Validators.required]],
      confpassword: [null, Validators.required]
    });
  }

getUsers (): void {
  this.projetService.getUsers().subscribe(users => (this.users = users))
}

confirm(username: string,firstname:string,lastname:string,email:string,address:string,country:string,city:string,phone:string,password:string,confpassword:string): void {
  
  this.projetService
  .getUsers()
  .subscribe((data: User[]) => {
    this.users = data;
    this.users.forEach(n => {
    if( confpassword == password && n.username == username && n.firstname == firstname && n.lastname == lastname && n.email == email && n.address == address && n.country == country && n.city == city && n.num_phone == phone){
      
      const newTask: User = { userId: n.userId , username:username, firstname:firstname , lastname:lastname , email:email , password:password , address:address , country:country , city:city , num_phone:phone ,image:""} as User
      this.projetService.updateuser(newTask).subscribe(() => this.getUsers())
      this._router.navigate(['/app-login'])
      Swal.fire({
        icon: 'success',
        title: 'Done',
        showConfirmButton: false,
        timer: 1500
      })
      
    }
    /*else {
      Swal.fire({
        icon: 'error',
        title: 'Something wrong!',
        text: 'Try again'
      })
    }*/
  })
    })
    
  }

}