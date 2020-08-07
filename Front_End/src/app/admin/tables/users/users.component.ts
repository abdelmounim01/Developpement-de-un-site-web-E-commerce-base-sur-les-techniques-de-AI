import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'
import { User } from '../../../service/Interfaces/user'
import { ProjetService } from '../../../service/ImplementationService/projet.service'

@Component({
  selector: 'app-users',
  templateUrl: './users.component.html',
  styleUrls: ['./users.component.css']
})
export class UsersComponent implements OnInit {
  users:User[]
  p: number=1;
  searchuser: string
  constructor(private projetService: ProjetService, private http: HttpClient) { }

  ngOnInit(): void {
    this.getUsers()
  
  }
  getUsers (): void {
    this.projetService.getUsers().subscribe(users => (this.users = users))
  }
  deleteuser (user: User): void {
    this.users = this.users.filter(h => h !== user)
    this.projetService
        .deleteUser(user.userId)
        .subscribe(() => console.log('User Deleted'))
  }
}
