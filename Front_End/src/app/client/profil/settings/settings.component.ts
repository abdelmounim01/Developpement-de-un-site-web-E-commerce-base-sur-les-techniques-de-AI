import { Component, OnInit , ViewChild } from '@angular/core';
import { User } from 'src/app/service/Interfaces/user';
import { ProjetService } from '../../../service/ImplementationService/projet.service'
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { HttpClient } from '@angular/common/http'
import { Router } from '@angular/router'

@Component({
  selector: 'app-settings',
  templateUrl: './settings.component.html',
  styleUrls: ['./settings.component.css']
})
export class SettingsComponent implements OnInit {
  users:User[]
  user:User
  form: FormGroup;
  id:number = JSON.parse(localStorage.getItem('user')).userId
  constructor(private projetService: ProjetService, private _router: Router, private http: HttpClient,private formBuilder: FormBuilder) { }

  ngOnInit(): void {
    this.getUsers();
    this.getUser(this.id);
    
    this.form = this.formBuilder.group({
      email: [null, [Validators.required]],
      password: [null, [Validators.required,Validators.minLength(6)]],
      username: [null, Validators.required],
      firstname: [null, Validators.required],
      lastname: [null, Validators.required],
      address: [null, Validators.required],
      num_phone: [null, Validators.required],
      city: [null, Validators.required],
      country: [null, Validators.required],
      file: [null, Validators.required],
    });
  }
  @ViewChild('fileInput') fileInput;
  uploadFile() {
    const files: FileList = this.fileInput.nativeElement.files;
    if (files.length === 0) {
      return;
    };
    this.projetService.parseTable(files).subscribe((data: any) => {
      console.log(data);
    });
}

getUsers (): void {
  this.projetService.getUsers().subscribe(users => (this.users = users))
}

getUser(id) : void{
  this.projetService.getUser(id).subscribe(user => (this.user = user))
}

updateuser(username: string,firstname:string,lastname:string,email:string,password:string,address:string,country:string,city:string,phone:string,fileInput:string): void {
  
  
  this.projetService
  .getUsers()
  .subscribe((data: User[]) => {
    this.users = data;
  const newTask: User = { userId: this.id , username:username, firstname:firstname , lastname:lastname , email:email , password:password , address:address , country:country , city:city , num_phone:phone , image:fileInput.replace("C:\\fakepath\\", "") } as User
  this.projetService.updateuser(newTask).subscribe(() => this.getUsers())
  this._router.navigate(['/app-profil'])
  })
}


}
