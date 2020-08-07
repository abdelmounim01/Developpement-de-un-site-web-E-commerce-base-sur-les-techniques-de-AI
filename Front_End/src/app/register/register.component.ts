import { Component, OnInit, ViewChild , HostBinding } from '@angular/core';
import { ProjetService } from '../service/ImplementationService/projet.service'
import { Router } from '@angular/router'
import { User } from '../service/Interfaces/user';
import { FormGroup, FormControl , Validators } from '@angular/forms';
import { AlertsService } from 'angular-alert-module';
import {  useAnimation } from '@angular/animations';
import { bounce } from 'ng-animate';
import { trigger,state,style,animate,transition}from '@angular/animations';



@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css'],
  /*
  animations: [
    trigger('InputError', [
      state('Normale', style({
        
      })),
      state('Error', style({
     
      })),
      transition('Normale => Error', [
        animate('3s')
      ]),
      
    ]),
  ],   */
})
export class RegisterComponent implements OnInit {
  exist:any
  users: User[]
  form: FormGroup;
  emailPattern = "^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}$";
  phoneNumber = "^(\+\d{1,3}[- ]?)?\d{10}$";
  bounce: any;

  
  constructor(private _auth: ProjetService,private _router: Router , private alerts: AlertsService) {
   }

  ngOnInit() {
    this.form = new FormGroup({
      username: new FormControl() ,
      email : new FormControl('' , Validators.required) ,
      password:new FormControl() ,
      firstname:new FormControl() ,
      lastname: new FormControl(),
      address: new FormControl(),
      num_phone:new FormControl(),
      city: new FormControl(),
      country: new FormControl(),
      type:new FormControl(),
      file: new FormControl(),
    });
  }

  @ViewChild('fileInput') fileInput;
  uploadFile() {
    const files: FileList = this.fileInput.nativeElement.files;
    if (files.length === 0) {
      return;
    };
}

  registerUser(Username: string , Email: string,Password:string,Address:string , Num_phone:string , City:string , Country:string ,Firstname:string,Lastname:string,Type:string,fileInput:string) {
    Email = Email.trim()
    Password= Password.trim()
    Username= Username.trim()

    this._auth
      .getUsers()
      .subscribe((data: User[]) => {
        this.users = data;
        this.users.forEach(n => {
          if(n.email == Email)
          {
            this.exist = 1
          }
          
        });
           if(this.exist == 1){
                this.alerts.setMessage('Email has already exist !','error');    
                this.alerts.setDefaults('timeout',5);
           }
           else{

          const newTask: User = { userId: this.users.length + 1 ,username:Username , email:Email,password:Password , address:Address , num_phone:Num_phone , city:City , country:Country,firstname:Firstname,lastname:Lastname,type:Type,image:fileInput.replace("C:\\fakepath\\", "")} as User
          this._auth.addUser(newTask).subscribe(() => this._auth.getUsers())
          this._router.navigate(['/app-login'])
       


        }
        this.exist = 0
  
        });

}


gotoTop() {
  window.scroll({ 
    top: 0, 
    left: 0, 
    behavior: 'smooth' 
  });
}


}
